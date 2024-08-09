# Author 
# Rahul Kumar (Northeastern University)

import pandas as pd
from ALBEF.CXR_ReDonE_module import RETRIEVAL_MODULE
import argparse

class argument():
    def __init__(self):
        self.impressions_path = '../../Dataset/Open-i/trainReport.csv'
        self.img_path = 'inference_images/'
        self.save_path = 'ALBEF/top_k_pred.csv'
        self.use_ve = False
        self.albef_retrieval_config = 'ALBEF/configs/Retrieval_flickr.yaml'
        self.albef_retrieval_ckpt = 'ALBEF/Pretrain/checkpoint_60.pth'
        self.albef_retrieval_top_k = 3
        self.albef_ve_config = 'configs/VE.yaml'
        self.albef_ve_ckpt = 'output/ve/checkpoint_7.pth'
        self.albef_ve_top_k = 10


def main():
    args = argument()
    df = pd.read_csv(args.impressions_path)
    impressions = df["findings"]
    cosine_sim_module = RETRIEVAL_MODULE(impressions=impressions, 
                                         mode='cosine-sim', 
                                         config=args.albef_retrieval_config, 
                                         checkpoint=args.albef_retrieval_ckpt, 
                                         topk=args.albef_retrieval_top_k,
                                         input_resolution=256, 
                                         img_path=args.img_path, 
                                         delimiter='|', 
                                         max_token_len = 25)
    cosine_sim_output = cosine_sim_module.predict()

    if args.use_ve:
        new_impressions = [el.split('|') for el in cosine_sim_output['Report Impression']]
    
        ve_module = RETRIEVAL_MODULE(impressions=new_impressions, 
                                                mode='visual-entailment', 
                                                config=args.albef_ve_config, 
                                                checkpoint=args.albef_ve_ckpt,
                                                topk=args.albef_ve_top_k,
                                                input_resolution=384, 
                                                img_path=args.img_path, 
                                                delimiter='[SEP]', 
                                                max_token_len=30)
        ve_output = ve_module.predict()
        ve_output.to_csv(args.save_path, index = False)
    else:
        cosine_sim_output.to_csv(args.save_path, index = False)

if __name__ == '__main__':
    main()