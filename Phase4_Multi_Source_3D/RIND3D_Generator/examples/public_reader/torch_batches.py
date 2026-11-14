import argparse
from rind3d_reader import RINDDataset,make_dataloader

def main():
 p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
 with RINDDataset(a.root,split=None) as data:
  loader=make_dataloader(data,batch_size=2,num_workers=a.workers,seed=42)
  for i,batch in enumerate(loader):
   print(batch['window_size'],tuple(batch['observed_response'].shape),batch['response_format'])
   assert 'candidate_sources' not in batch
   if i==2:break
if __name__=='__main__':main()
