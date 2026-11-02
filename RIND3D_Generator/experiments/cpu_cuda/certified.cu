// All accepted predicates use directed binary64 intervals. No epsilon.
struct I { double lo,hi; };
__device__ I add(I a,I b){return {__dadd_rd(a.lo,b.lo),__dadd_ru(a.hi,b.hi)};}
__device__ I product(double a,double b){return {__dmul_rd(a,b),__dmul_ru(a,b)};}
// Coordinates are certified nonnegative and <=1. Coefficients are exact
// rational enclosures transported as [lower,upper] binary64 endpoints.
__device__ I axis(I a,double lo,double hi){
 if(a.lo==0. && a.hi==0.)return {0.,0.};
 if(a.lo>=0.)return {__dmul_rd(a.lo,lo),__dmul_ru(a.hi,hi)};
 if(a.hi<=0.)return {__dmul_rd(a.lo,hi),__dmul_ru(a.hi,lo)};
 return {__dmul_rd(a.lo,hi),__dmul_ru(a.hi,hi)};
}
__device__ int body(const double*rows,int begin,int end,const double*lo,const double*hi){
 bool inside=true;
 for(int r=begin;r<end;++r){const double*f=rows+8*r;I v={f[6],f[7]};
  for(int a=0;a<3;++a)v=add(v,axis({f[2*a],f[2*a+1]},lo[a],hi[a]));
  if(!isfinite(v.lo)||!isfinite(v.hi))return -1;
  if(v.lo>0.)return 0; inside=inside && v.hi<=0.;
 }return inside?1:-1;
}
__device__ int united(const double*rows,const int*offset,int bodies,const double*lo,const double*hi){
 bool unknown=false;for(int b=0;b<bodies;++b){int v=body(rows,offset[b],offset[b+1],lo,hi);if(v==1)return 1;unknown=unknown||v<0;}return unknown?-1:0;
}
extern "C" __global__ void regions(const double*g,const int*go,const double*s,const int*so,int bodies,
 const int*origins,int batches,int side,int n,int level,signed char*labels,int skip_mask){
 int dim=1<<level, nodes=dim*dim*dim, b=blockIdx.y,k=blockIdx.x*blockDim.x+threadIdx.x;
 if(b>=batches||k>=nodes)return;
 int x=k%dim,y=(k/dim)%dim,z=k/(dim*dim),offset=(nodes-1)/7;
 int cy=-1,cm=skip_mask?0:-1;
 if(level){int pd=dim/2,pn=nodes/8,po=(pn-1)/7;
  int parent=po+x/2+pd*(y/2+pd*(z/2));cy=labels[2*(b*585+parent)];if(!skip_mask)cm=labels[2*(b*585+parent)+1];
 }
 int size=side/dim; double lo[3],hi[3];int xyz[3]={x,y,z};
 for(int a=0;a<3;++a){int o=origins[3*b+a]+xyz[a]*size;lo[a]=(o+.5)/n;hi[a]=(o+size-.5)/n;}
 if(cm<0)cm=united(g,go,bodies,lo,hi);
 if(cm==1)cy=0;
 if(cy<0){int blocked=united(s,so,bodies,lo,hi);cy=blocked<0?-1:1-blocked;}
 labels[2*(b*585+offset+k)]=cy; labels[2*(b*585+offset+k)+1]=cm;
}
extern "C" __global__ void packed(const double*g,const int*go,const double*s,const int*so,int bodies,
 const int*origins,int batches,int side,int n,int leaf_level,const signed char*labels,
 unsigned char*yout,unsigned char*mout,unsigned char*uyout,unsigned char*umout,int skip_mask){
 int b=blockIdx.y,k=blockIdx.x*blockDim.x+threadIdx.x;if(b>=batches||k>=32768)return;
 int x=(k%8)*8,y=(k/8)%64,z=k/512;int at=b*32768+k;
 unsigned char yy=0,mm=0,uy=0,um=0;
 if(x<side&&y<side&&z<side){int dim=1<<leaf_level,off=(dim*dim*dim-1)/7;
  int leaf=off+x/8+dim*(y/8+dim*(z/8));int cy=labels[2*(b*585+leaf)],cm=skip_mask?0:labels[2*(b*585+leaf)+1];
  for(int i=0;i<8;++i){double p[3]={(origins[3*b]+x+i+.5)/n,(origins[3*b+1]+y+.5)/n,(origins[3*b+2]+z+.5)/n};
   int m=cm,yv=cy;if(m<0)m=united(g,go,bodies,p,p);
   if(yv<0){if(m==1)yv=0;else{int v=united(s,so,bodies,p,p);yv=v<0?-1:1-v;}}
   yy|=(yv==1)<<i;mm|=(m==1)<<i;uy|=(yv<0)<<i;um|=(m<0)<<i;
  }
 }yout[at]=yy;mout[at]=mm;uyout[at]=uy;umout[at]=um;
}
extern "C" __global__ void points(const double*g,const int*go,const double*s,const int*so,int bodies,
 const double*p,int count,signed char*y,signed char*m){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 m[i]=united(g,go,bodies,p+3*i,p+3*i);int v=united(s,so,bodies,p+3*i,p+3*i);y[i]=v<0?-1:1-v;
}
extern "C" __global__ void arithmetic(const double*a,const double*b,int count,double*out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 out[4*i]=__dadd_rd(a[i],b[i]);out[4*i+1]=__dadd_ru(a[i],b[i]);
 out[4*i+2]=__dmul_rd(a[i],b[i]);out[4*i+3]=__dmul_ru(a[i],b[i]);
}
