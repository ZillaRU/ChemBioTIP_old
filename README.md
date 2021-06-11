# BioMIP
Abstract

### Preprocess
接触图预处理

1. 查看物理CPU个数
   `cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l`
   
    查看每个物理CPU中core的个数(即核数)
   
   `cat /proc/cpuinfo| grep "cpu cores"| uniq`
   
   查看逻辑CPU的个数 
   
   `cat /proc/cpuinfo| grep "processor"| wc -`
   
    查看内存信息
   
    `cat /proc/meminfo`
2.
    ```
    top -c
    ps  -H -efwwww
    watch -n 5 nvidia-smi
    find xxxx/xxxxx -type f -print | wc -l
    ```
3. 
    🔥cpu 内存
    ```
    sudo nohup python msa_aln_gen.py xxxxx >/dev/null 2>&1 &
   ```

    🔥gpu 
   ```
   conda activate py37cmap
    nohup python3 cmap_gen.py xxxxx  0/1/2 >/dev/null 2>&1 &
   ```

> 使用nohup时，会自动将输出写入nohup.out文件中，nohup.out不停的增大，可以利用`/dev/null`来解决这个问题。
- 只保存错误信息
  `nohup【cmd】>/dev/null 2>log &`
    
- 错误信息也不想要
  `nohup【cmd】>/dev/null 2>&1 &`



### Requirements
- 
- pytorch==
- dgl==
- dgllife==


