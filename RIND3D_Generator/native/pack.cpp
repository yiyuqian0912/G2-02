#include "pack.hpp"
#include <zstd.h>
#include <algorithm>
#include <bit>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <unistd.h>

namespace r3 {
static void put(std::string& b,std::size_t p,std::uint64_t v,std::size_t n) {
 for(std::size_t k=0;k<n;++k)b[p+k]=char((v>>(8*k))&255);
}
static std::uint64_t get(const std::string& b,std::size_t p,std::size_t n) {
 std::uint64_t v=0;for(std::size_t k=0;k<n;++k)v|=std::uint64_t(std::uint8_t(b[p+k]))<<(8*k);return v;
}
std::uint32_t crc(const std::string& s) {
 static const auto table=[] {std::array<std::uint32_t,256> t{};for(unsigned i=0;i<256;++i){auto c=i;for(int k=0;k<8;++k)c=(c>>1)^((c&1)?0xedb88320U:0);t[i]=c;}return t;}();
 auto c=0xffffffffU;for(unsigned char x:s)c=(c>>8)^table[(c^x)&255];return c^0xffffffffU;
}
Encoded encode(const std::string& raw,int level) {
 if(raw.size()!=RAW)throw std::invalid_argument("A brick must contain exactly 32768 packed bytes");
 std::uint32_t ones=0;for(unsigned char c:raw)ones+=std::popcount(c);
 Encoded e{2,raw,crc(raw),ones};
 if(ones==0){e.codec=0;e.payload.clear();return e;}
 if(ones==RAW*8){e.codec=1;e.payload.clear();return e;}
 std::string compressed(ZSTD_compressBound(RAW),'\0');
 auto n=ZSTD_compress(compressed.data(),compressed.size(),raw.data(),RAW,level);
 if(ZSTD_isError(n))throw std::runtime_error(ZSTD_getErrorName(n));
 if(n<RAW){compressed.resize(n);e.codec=3;e.payload=std::move(compressed);}return e;
}
static void seek(std::FILE* f,std::uint64_t p){if(fseeko(f,off_t(p),SEEK_SET))throw std::runtime_error("seek failed");}
static std::string read(std::FILE* f,std::size_t n){std::string s(n,'\0');if(std::fread(s.data(),1,n,f)!=n)throw std::runtime_error("Truncated RINDPack");return s;}
static void write_bytes(std::FILE* f,const std::string& s){if(std::fwrite(s.data(),1,s.size(),f)!=s.size())throw std::runtime_error("RINDPack write failed");}
void Writer::header(bool complete){
 std::string h(HEADER,'\0');h.replace(0,8,"RINDB001",8);put(h,8,1,2);put(h,10,HEADER,2);put(h,12,complete,4);
 for(int a=0;a<3;++a){put(h,16+4*a,shape_[a],4);put(h,28+2*a,64,2);}
 put(h,36,RECORD,2);put(h,38,1,2);put(h,40,done_.size(),4);put(h,44,HEADER,8);
 put(h,52,HEADER+done_.size()*RECORD,8);put(h,60,complete?tail_:0,8);
 h.replace(68,32,identity_);h.replace(100,32,policy_);put(h,252,crc(h.substr(0,252)),4);
 seek(file_,0);write_bytes(file_,h);
}
Writer::Writer(std::string path,std::array<std::uint32_t,3> shape,std::string identity,
 std::string policy,bool resume,int level):path_(std::move(path)),partial_(path_+".partial"),identity_(std::move(identity)),policy_(std::move(policy)),shape_(shape),level_(level) {
 try {
 if(identity_.size()!=32||policy_.size()!=32)throw std::invalid_argument("Identity and policy require 32-byte hashes");
 std::uint64_t count=1;for(auto n:shape_){if(n==0||n>1024)throw std::invalid_argument("Shape components must be 1..1024");count*=(n+63)/64;}
 done_.resize(count,false);tail_=HEADER+count*RECORD;
 if(std::filesystem::exists(path_))throw std::runtime_error("Final file already exists");
 if(std::filesystem::exists(partial_)){
  if(!resume)throw std::runtime_error("Partial file exists; explicitly request resume");
  file_=std::fopen(partial_.c_str(),"r+b");if(!file_)throw std::runtime_error("Cannot open partial");
  auto h=read(file_,HEADER);
  if(h.substr(0,8)!="RINDB001"||get(h,8,2)!=1||get(h,10,2)!=HEADER||get(h,252,4)!=crc(h.substr(0,252))||h.substr(68,32)!=identity_||h.substr(100,32)!=policy_)throw std::runtime_error("Resume header/identity/policy mismatch");
  for(int a=0;a<3;++a)if(get(h,16+4*a,4)!=shape_[a])throw std::runtime_error("Resume shape mismatch");
  if(get(h,12,4)>1||get(h,34,1)||get(h,35,1)||get(h,36,2)!=RECORD||get(h,38,2)!=1||get(h,40,4)!=count||get(h,44,8)!=HEADER||get(h,52,8)!=tail_)throw std::runtime_error("Resume format policy mismatch");
  for(int a=0;a<3;++a)if(get(h,28+2*a,2)!=64)throw std::runtime_error("Resume brick shape mismatch");
  for(int a=132;a<252;++a)if(h[a])throw std::runtime_error("Resume reserved header bytes");
  const auto bytes=std::filesystem::file_size(partial_);
  if(bytes<tail_||(get(h,12,4)&&get(h,60,8)!=bytes))throw std::runtime_error("Resume file size mismatch");
  std::vector<std::pair<std::uint64_t,std::uint64_t>> ranges;
  for(std::size_t i=0;i<count;++i){
   seek(file_,HEADER+i*RECORD);auto r=read(file_,RECORD);if(!std::uint8_t(r[13]))continue;
   auto offset=get(r,0,8),length=get(r,8,4),codec=get(r,12,1);
   if(get(r,13,1)!=1||get(r,14,2)||get(r,28,4)||codec>3||length>RAW||get(r,24,4)>RAW*8||(length&&(offset<tail_||offset+length>bytes)))throw std::runtime_error("Invalid resume chunk index");
   if(codec<2&&(offset||length||get(r,20,4)||get(r,24,4)!=(codec?RAW*8:0)))throw std::runtime_error("Invalid resume uniform brick");
   if(codec>=2&&!length)throw std::runtime_error("Empty resume payload");
   if(length)ranges.emplace_back(offset,offset+length);
   std::string stored;if(length){seek(file_,offset);stored=read(file_,length);}
   if(crc(stored)!=get(r,20,4))throw std::runtime_error("Resume stored checksum mismatch");
   std::string raw(RAW,codec==1?char(255):0);
   if(codec==2){if(length!=RAW)throw std::runtime_error("Invalid raw length");raw=stored;}
   if(codec==3){auto n=ZSTD_decompress(raw.data(),RAW,stored.data(),stored.size());if(ZSTD_isError(n)||n!=RAW)throw std::runtime_error("Invalid compressed resume brick");}
   if(crc(raw)!=get(r,16,4))throw std::runtime_error("Resume decoded checksum mismatch");
   std::uint32_t ones=0;for(unsigned char c:raw)ones+=std::popcount(c);
   if(ones!=get(r,24,4))throw std::runtime_error("Resume population count mismatch");
   done_[i]=true;
  }
  std::sort(ranges.begin(),ranges.end());
  for(std::size_t i=1;i<ranges.size();++i)if(ranges[i].first<ranges[i-1].second)throw std::runtime_error("Overlapping resume payloads");
  tail_=bytes;
 }else{
  file_=std::fopen(partial_.c_str(),"w+b");if(!file_)throw std::runtime_error("Cannot create partial file");
  header(false);write_bytes(file_,std::string(count*RECORD,'\0'));checkpoint();
 }
 }catch(...){if(file_)std::fclose(file_);file_=nullptr;throw;}
}
Writer::~Writer(){if(file_)std::fclose(file_);}
std::vector<std::size_t> Writer::committed()const{std::vector<std::size_t> r;for(std::size_t i=0;i<done_.size();++i)if(done_[i])r.push_back(i);return r;}
void Writer::write(std::size_t index,const std::string& raw){
 if(finished_||!file_||index>=done_.size()||done_[index])throw std::invalid_argument("Invalid or duplicate brick index");
 auto e=encode(raw,level_);std::string r(RECORD,'\0');
 if(!e.payload.empty()){put(r,0,tail_,8);seek(file_,tail_);write_bytes(file_,e.payload);tail_+=e.payload.size();}
 put(r,8,e.payload.size(),4);put(r,12,e.codec,1);put(r,13,1,1);put(r,16,e.checksum,4);put(r,20,crc(e.payload),4);put(r,24,e.ones,4);
 seek(file_,HEADER+index*RECORD);write_bytes(file_,r);done_[index]=true;
}
void Writer::checkpoint(){if(!file_)return;if(std::fflush(file_)||fsync(fileno(file_)))throw std::runtime_error("Checkpoint flush failed");}
void Writer::finish(){
 if(finished_||std::find(done_.begin(),done_.end(),false)!=done_.end())throw std::runtime_error("Cannot publish an incomplete RINDPack");
 // A crash can leave orphan payload bytes. Compact them before publication so
 // the raw payload bound remains true after arbitrary valid resumptions.
 std::vector<std::string> records;std::uint64_t sum=0;
 for(std::size_t i=0;i<done_.size();++i){seek(file_,HEADER+i*RECORD);records.push_back(read(file_,RECORD));sum+=get(records.back(),8,4);}
 const auto data=HEADER+done_.size()*RECORD;
 if(tail_!=data+sum){
  auto compact=partial_+".compact";auto other=std::fopen(compact.c_str(),"w+b");if(!other)throw std::runtime_error("Cannot compact recovery file");
  seek(file_,0);write_bytes(other,read(file_,HEADER));write_bytes(other,std::string(done_.size()*RECORD,'\0'));std::uint64_t cursor=data;
  for(std::size_t i=0;i<records.size();++i){auto&r=records[i];auto n=get(r,8,4);if(n){seek(file_,get(r,0,8));auto bytes=read(file_,n);seek(other,cursor);write_bytes(other,bytes);put(r,0,cursor,8);cursor+=n;}seek(other,HEADER+i*RECORD);write_bytes(other,r);}
  if(std::fflush(other)||fsync(fileno(other)))throw std::runtime_error("Compaction flush failed");std::fclose(file_);std::fclose(other);
  std::filesystem::rename(compact,partial_);file_=std::fopen(partial_.c_str(),"r+b");if(!file_)throw std::runtime_error("Compaction reopen failed");tail_=cursor;
 }
 header(true);checkpoint();std::fclose(file_);file_=nullptr;
 std::filesystem::rename(partial_,path_);finished_=true;
}
}
