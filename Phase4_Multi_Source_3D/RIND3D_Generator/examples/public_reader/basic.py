import argparse
from rind3d_reader import RINDDataset

def main():
 p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args()
 with RINDDataset(a.root,split=None) as data:
  print('scenes',len(data.scenes),'windows',len(data))
  item=data[0];scene=data.scene(item['scene_id']);counts=scene.counts(tuple(item['window_origin']),item['window_size'])
  print(item['scene_id'],item['window_origin'],item['window_size'],counts)
  print('unaligned solid ROI',scene.roi((17,29,43),79,field='solid',packed=False).shape)
if __name__=='__main__':main()
