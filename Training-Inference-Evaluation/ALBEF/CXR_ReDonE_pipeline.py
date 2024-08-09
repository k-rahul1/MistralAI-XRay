import pandas as pd
from CXR_ReDonE_module import RETRIEVAL_MODULE
import argparse

def main(args):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--impressions_path', default='../../Dataset/Open-i/trainReport.csv', help= 'path to the open-i train corpus')
    parser.add_argument('--img_path', default='../../Dataset/Open-i/test/', help = 'path to the test set image folder') 
    parser.add_argument('--save_path', default='ALBEF/top_k_pred.csv')
    parser.add_argument('--use_ve', type = bool, default=False)
    parser.add_argument('--albef_retrieval_config', default='ALBEF/configs/Retrieval_flickr.yaml')
    parser.add_argument('--albef_retrieval_ckpt', default='ALBEF/Pretrain/checkpoint_60.pth')
    parser.add_argument('--albef_retrieval_top_k', type = int, default=50)
    parser.add_argument('--albef_ve_config', default='configs/VE.yaml')
    parser.add_argument('--albef_ve_ckpt', default='output/ve/checkpoint_7.pth')    
    parser.add_argument('--albef_ve_top_k', default=10, type = int, help='url used to set up distributed training')
    args = parser.parse_args()

    if not args.use_ve:
        args.albef_retrieval_top_k = 3
    
    main(args)