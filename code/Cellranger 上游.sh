export http_proxy=http://10.255.128.100:3128
export https_proxy=http://10.255.128.100:3128
export ftp_proxy=http://10.255.128.100:3128

## cellranger单细胞分析流程主要分为：
## 数据拆分 cellranger mkfastq
## 细胞定量 cellranger count
## 组合分析 cellranger aggr
## 参数调整 cellranger reanalyze
## 还有一些用户可能会用到的功能：mat2csv、mkgtf、mkref、vdj、mkvdjref、testrun、upload、sitecheck。


##### 安装 Cell Ranger ##### 
mkdir yard
pwd
cd /mnt/home/user.name/yard
mkdir apps
cd apps

curl -o cellranger-6.1.2.tar.gz "https://cf.10xgenomics.com/releases/cell-exp/cellranger-6.1.2.tar.gz?Expires=1641366506&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9jZi4xMHhnZW5vbWljcy5jb20vcmVsZWFzZXMvY2VsbC1leHAvY2VsbHJhbmdlci02LjEuMi50YXIuZ3oiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2NDEzNjY1MDZ9fX1dfQ__&Signature=kaV8~ZabHhyDykUhbN~F78PDQfNZ64IamgsGc1nOSghFKPr0fbZ3WJk-2eWYh7IEt-KupenYP89W1zHi4lrxF~ZBbuP4NTaKEAa-G6ILJoX-VdyFnktkXFYDHgzEJ8ABq-NM6RWn20WD3a9BITNHTIWPtxjM-NaXAuR5uc5PuAEgjSDaQ2QBAQr~1q4aSM-~vJt~ia5e8acTz9RlM24EluLqfO59VCtAorP-5iJRwvLw9DjfrTlDtWfy3M2LSXp5OGmVJH1WUQReLK~0iZX2e8~vrHlAYpuxMa0Lgil6oHQ5s6vc~Dod3Aqpjb9sM~wuVo80zi4EqJ5nq0LU8SNbiQ__&Key-Pair-Id=APKAI7S6A5RYOXBWRPDA"

wget -O cellranger-6.1.2.tar.gz "https://cf.10xgenomics.com/releases/cell-exp/cellranger-6.1.2.tar.gz?Expires=1641366506&Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9jZi4xMHhnZW5vbWljcy5jb20vcmVsZWFzZXMvY2VsbC1leHAvY2VsbHJhbmdlci02LjEuMi50YXIuZ3oiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2NDEzNjY1MDZ9fX1dfQ__&Signature=kaV8~ZabHhyDykUhbN~F78PDQfNZ64IamgsGc1nOSghFKPr0fbZ3WJk-2eWYh7IEt-KupenYP89W1zHi4lrxF~ZBbuP4NTaKEAa-G6ILJoX-VdyFnktkXFYDHgzEJ8ABq-NM6RWn20WD3a9BITNHTIWPtxjM-NaXAuR5uc5PuAEgjSDaQ2QBAQr~1q4aSM-~vJt~ia5e8acTz9RlM24EluLqfO59VCtAorP-5iJRwvLw9DjfrTlDtWfy3M2LSXp5OGmVJH1WUQReLK~0iZX2e8~vrHlAYpuxMa0Lgil6oHQ5s6vc~Dod3Aqpjb9sM~wuVo80zi4EqJ5nq0LU8SNbiQ__&Key-Pair-Id=APKAI7S6A5RYOXBWRPDA"

ls -l
# cellranger-6.1.2.tar.gz

tar -zxvf cellranger-6.1.2.tar.gz
# cellranger-6.1.2/
# cellranger-6.1.2/.env.json
# cellranger-6.1.2/.version
# cellranger-6.1.2/LICENSE
# cellranger-6.1.2/builtwith.json
# cellranger-6.1.2/sourceme.bash
# cellranger-6.1.2/sourceme.csh
# cellranger-6.1.2/bin/
# cellranger-6.1.2/bin/_cellranger_internal
# cellranger-6.1.2/bin/cellranger
# cellranger-6.1.2/bin/rna/
# ...

ls -1
# cellranger-6.1.2
# cellranger-6.1.2.tar.gz

cd cellranger-6.1.2
pwd
# /mnt/home/user.name/yard/apps/cellranger-6.1.2

export PATH=/mnt/home/user.name/yard/apps/cellranger-6.1/bin:$PATH

which cellranger
# ~/yard/apps/cellranger-6.1.2/cellranger

cellranger
cellranger sitecheck > sitecheck.txt
less sitecheck.txt
cellranger testrun --id=check_install



##### Running cellranger mkfastq #####

mkdir ~/yard/run_cellranger_mkfastq
cd ~/yard/run_cellranger_mkfastq

wget https://cf.10xgenomics.com/supp/cell-exp/cellranger-tiny-bcl-1.2.0.tar.gz
wget https://cf.10xgenomics.com/supp/cell-exp/cellranger-tiny-bcl-simple-1.2.0.csv
tar -zxvf cellranger-tiny-bcl-1.2.0.tar.gz

tree -L 2 cellranger-tiny-bcl-1.2.0/

cellranger mkfastq --id=mkfastq --run=/mnt/home/user.name/yard/10X_data/190322_A00111_0033_AH3Y5YDMXX --csv=/mnt/home/user.name/yard/10X_data/190322_A00111_0033_AH3Y5YDMXX/SampleSheet.csv

ls -altR cellranger-tiny-bcl-1.2.0/

cellranger mkfastq --help

cellranger mkfastq --id=tutorial_walk_through \
  --run=run_cellranger_mkfastq/cellranger-tiny-bcl-1.2.0 \
  --csv=run_cellranger_mkfastq/cellranger-tiny-bcl-simple-1.2.0.csv

cd /mnt/home/user.name/yard/run_cellranger_mkfastq/tutorial_walk_through/outs/fastq_path
ls -1

ls -1 H35KCBCXY/test_sample



##### Running cellranger count #####
mkdir ~/yard/run_cellranger_count
cd ~/yard/run_cellranger_count

wget https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_1k_v3/pbmc_1k_v3_fastqs.tar

tar -xvf pbmc_1k_v3_fastqs.tar

wget https://cf.10xgenomics.com/supp/cell-exp/refdata-gex-GRCh38-2020-A.tar.gz
tar -zxvf refdata-gex-GRCh38-2020-A.tar.gz

cellranger count --help

cellranger count --id=run_count_1kpbmcs \
   --fastqs=run_cellranger_count/pbmc_1k_v3_fastqs \
   --sample=pbmc_1k_v3 \
   --transcriptome=refdata-gex-GRCh38-2020-A

ls -1 run_count_1kpbmcs/outs






