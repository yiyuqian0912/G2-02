"""No generation in the loader. Use --diagnostic only for INCOMPLETE examples."""
import argparse
from rind3d import RINDDataset
from rind3d.data import make_dataloader


def main():
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--workers',type=int,default=2)
    p.add_argument('--diagnostic',action='store_true');a=p.parse_args()
    with RINDDataset(a.root,split='train',require_ready=not a.diagnostic,response_format='bool') as ds:
        print('Stored windows:',len(ds))
        for i,batch in enumerate(make_dataloader(ds,batch_size=2,num_workers=a.workers)):
            print(batch['scene_id'],batch['window_id'].tolist(),tuple(batch['observed_response'].shape),batch['response_format'])
            if i>=3:break


if __name__=='__main__':main()
