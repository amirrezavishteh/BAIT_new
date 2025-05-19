"""
main.py: Main entry point for the BAIT (LLM Backdoor Scanning) project.

Author: [NoahShen]
Organization: [PurduePAML]
Date: [2024-09-25]
Version: 1.0

This module serves as the main entry point for the BAIT project. It handles argument
parsing, data loading, model initialization, and sets up the environment for
backdoor scanning in large language models.

Copyright (c) [2024] [PurduePAML]
"""
import torch
from transformers import HfArgumentParser
from loguru import logger
import sys
from time import time
from bait.argument import BAITArguments, ModelArguments, DataArguments, ScanArguments
from bait.utils import seed_everything
from bait.model import build_model, parse_model_args
from bait.data.bait_extend import build_data_module
from bait.eval import Evaluator
from bait.constants import SEED
from transformers.utils import logging
import os
import json
from pprint import pprint
from bait.bait import BAIT


logging.get_logger("transformers").setLevel(logging.ERROR)


# TODO: some models fail to upload, check


seed_everything(SEED)

class BaitRunner:
    def __init__(self, scan_args):

        self.model_zoo_dir = scan_args.model_zoo_dir
        self.model_id = scan_args.model_id
        self.data_dir = scan_args.data_dir
        self.output_dir = scan_args.output_dir
        self.cache_dir = scan_args.cache_dir

        self.run_dir = os.path.join(scan_args.output_dir, scan_args.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        if self.model_id == "":
            self.model_idxs = [f for f in os.listdir(self.model_zoo_dir) if f.startswith("id-")]
            self.model_idxs.sort()

        else:
            self.model_idxs = [self.model_id]
        

        self.model_configs = []
        for model_idx in self.model_idxs:
            model_config_path = os.path.join(self.model_zoo_dir, f"{model_idx}", "config.json")
            with open(model_config_path, "r") as f:
                model_config = json.load(f)
            self.model_configs.append(model_config)
        

    def _init_args(self, model_id, model_config): 

        bait_args = BAITArguments()
        model_args = ModelArguments()
        data_args = DataArguments()
    
        # Add this check after parsing arguments
        if bait_args.warmup_batch_size > data_args.prompt_size:
            bait_args.warmup_batch_size = data_args.prompt_size
            logger.warning(f"warmup_batch_size was greater than prompt_size. Setting warmup_batch_size to {data_args.prompt_size}")
        
        if bait_args.uncertainty_inspection_times_threshold > bait_args.warmup_steps:
            bait_args.uncertainty_inspection_times_threshold = bait_args.warmup_steps
            logger.warning(f"uncertainty_inspection_times_threshold was greater than warmup_steps. Setting uncertainty_inspection_times_threshold to {bait_args.warmup_steps}")
        
        bait_args.batch_size = data_args.batch_size
        bait_args.prompt_size = data_args.prompt_size
        model_args, data_args = parse_model_args(model_config, data_args, model_args)
        model_args.adapter_path = os.path.join(self.model_zoo_dir, model_id, "model")
        model_args.cache_dir = self.cache_dir
        data_args.data_dir = self.data_dir

        return bait_args, model_args, data_args


    def _scan_a_model(self, model_id, model_config):

        bait_args, model_args, data_args = self._init_args(model_id, model_config)

        # log directory
        log_dir = os.path.join(self.run_dir, model_id)
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logging to both console and file
        log_file = os.path.join(log_dir, "scan.log")
        logger.remove()  # Remove default handler
        logger.add(sys.stderr, level="INFO")  # Add console handler
        logger.add(log_file, rotation="10 MB", level="DEBUG")  # Add file handler

        with open(os.path.join(log_dir, "arguments.json"), "w") as f:
            json.dump({"bait_args": vars(bait_args), "model_args": vars(model_args), "data_args": vars(data_args)}, f, indent=4)
        
        
        logger.info("BAIT Arguments:")
        pprint(vars(bait_args))
        
        logger.info("Model Arguments:")
        pprint(vars(model_args))
        
        logger.info("Data Arguments:")
        pprint(vars(data_args))
        
        # load model 
        logger.info("Loading model...")
        model, tokenizer = build_model(model_args)
        logger.info("Model loaded successfully")

        # load data
        logger.info("Loading data...")
        dataset, dataloader = build_data_module(data_args, tokenizer, logger)
        logger.info("Data loaded successfully")

        # initialize BAIT LLM backdoor scanner
        scanner = BAIT(model, tokenizer, dataloader, bait_args, logger, device = torch.device(f'cuda:{model_args.gpu}'))
        start_time = time()
        is_backdoor, q_score, invert_target = scanner.run()
        end_time = time()

        # Log the results
        logger.info(f"Is backdoor detected: {is_backdoor}")
        logger.info(f"Q-score: {q_score}")
        logger.info(f"Invert target: {invert_target}")
        logger.info(f"Time taken: {end_time - start_time} seconds")
        
        result = {
            "is_backdoor": is_backdoor,
            "q_score": q_score,
            "invert_target": invert_target,
            "time_taken": end_time - start_time
        }
        with open(os.path.join(log_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=4)
    

    def run(self):
        for model_id, model_config in zip(self.model_idxs, self.model_configs):

            # check if result.json exists
            result_path = os.path.join(self.run_dir, model_id, "result.json")
            if os.path.exists(result_path):
                logger.info(f"Result for model {model_id} already exists. Skipping...")
                continue
            try:
                self._scan_a_model(model_id, model_config)
            except Exception as e:
                logger.error(f"Error scanning model {model_id}: {e}")
                logger.exception("Exception occurred while scanning model")

        Evaluator().eval(self.run_dir)
    


if __name__ == "__main__":
    parser = HfArgumentParser((ScanArguments))
    scan_args = parser.parse_args_into_dataclasses()[0]  # Extract the first element from the tuple
    runner = BaitRunner(scan_args)
    runner.run()