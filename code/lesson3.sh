#!/bin/bash

ls -a # 列出目录所有文件，包含以.开始的隐藏文件
ls -A # 列出除.及..的其它文件
ls -r # 反序排列
ls -t # 以文件修改时间排序
ls -S # 以文件大小排序
ls -h # 以易读大小显示
ls -l # 除了文件名之外，还将文件的权限、所有者、文件大小等信息详细列出来

cd /  # 进入根目录
cd ~  # 进入 "home" 目录，等于cd命令（可以省略 ~ ）
cd -  # 进入上一次工作路径
cd !$ # 把上个命令的参数作为cd参数使用

pwd   # 命令用于查看当前工作目录路径

mkdir t # 当前工作目录下创建名为 t的文件夹
mkdir -p /tmp/test/t1/t # 在 tmp 目录下创建路径为 test/t1/t 的目录，若不存在，则创建：

rm -i *.log # 删除任何 .log 文件，删除前逐一询问确认
rm -rf test # 删除 test 子目录及子目录中所有档案删除，并且不用一一确认
rm -- -f*   # 删除以 -f 开头的文件
rmdir t     # 删除当前工作目录下名为 t 的文件夹

mv test.log test1.txt # 将文件 test.log 重命名为 test1.txt
mv log1.txt log2.txt log3.txt /test3/ # 将文件 log1.txt,log2.txt,log3.txt 移动到根的 test3 目录中
mv -i log1.txt log2.txt # 将文件 file1 改名为 file2，如果 file2 已经存在，则询问是否覆盖
mv * ../ # 移动当前文件夹下的所有文件到上一级目录

cp -ai a.txt test/ # 复制 a.txt 到 test 目录下，保持原文件时间，如果原文件存在提示是否覆盖。
cp -s a.txt link_a.txt # 为 a.txt 建立一个链接（快捷方式）

cat filename # 一次显示整个文件
cat file1 file2 > file # 将几个文件合并为一个文件
cat -n log2012.log > log2013.log # 把log2012.log 的文件内容加上行号后输入 log2013.log 这个文件里
cat -b log2012.log log2013.log > log.log # 把log2012.log 和 log2013.log 的文件内容加上行号（空白行不加）之后将内容附加到 log.log 里

more +3 text.txt  # 显示文件中从第3行起的内容
ls -l | more -5   # 在所列出文件目录详细信息，借助管道使每次显示 5 行
ps -aux | less -N # ps 查看进程信息并通过 less 分页显示
less 1.log 2.log  # 查看多个文件内容

head 1.log -n 20 # 显示 1.log 文件中前 20 行
head -c 20 log2014.log # 显示 1.log 文件前 20 字节
head -n -10 t.log # 显示 t.log最后 10 行
tail -f 1.log # 实时显示 1.log 文件的最后 10 行
ping 127.0.0.1 > ping.log & # 循环读取逐渐增加的文件内容
tail -f ping.log # 后台运行：可使用 jobs -l 查看，也可使用 fg 将其移到前台运行。

tar -c # 建立新的压缩文件
tar -f # 指定压缩文件
tar -r # 添加文件到已经压缩文件包中
tar -u # 添加改了和现有的文件到压缩包中
tar -x # 从压缩包中抽取文件
tar -t # 显示压缩文件中的内容
tar -z # 支持gzip压缩
tar -j # 支持bzip2压缩
tar -Z # 支持compress解压文件
tar -v # 显示操作过程

chown -c # 显示更改的部分的信息
chown -R # 处理指定目录及子目录下的所有文件
chown -c uesr:group log2012.log # 改变拥有者和群组,并显示改变信息
chown -c :mail t.log # 改变文件群组
chown -cR uesr: test/ # 改变文件夹及子文件目录属主及属组为 mail

which   #查看可执行文件的位置。
whereis #查看文件的位置。
locate  #配合数据库查看文件位置。
find    #实际搜寻硬盘查询文件名称。

find -name # 按照文件名查找文件
find -perm # 按文件权限查找文件
find -user # 按文件属主查找文件
find -group  # 按照文件所属的组来查找文件。
find -type  # 查找某一类型的文件，诸如：
   # b - 块设备文件
   # d - 目录
   # c - 字符设备文件
   # l - 符号链接文件
   # p - 管道文件
   # f - 普通文件
find -size n   # n:[c] 查找文件长度为n块文件，带有c时表文件字节大小
find -amin n   # 查找系统中最后N分钟访问的文件
find -atime n  # 查找系统中最后n*24小时访问的文件
find -cmin n   # 查找系统中最后N分钟被改变文件状态的文件
find -ctime n  # 查找系统中最后n*24小时被改变文件状态的文件
find -mmin n   # 查找系统中最后N分钟被改变文件数据的文件
find -mtime n  # 查找系统中最后n*24小时被改变文件数据的文件
# (用减号-来限定更改时间在距今n日以内的文件，而用加号+来限定更改时间在距今n日以前的文件。 )
find -maxdepth n # 最大查找目录深度
find -prune # 选项来指出需要忽略的目录。在使用-prune选项时要当心，因为如果你同时使用了-depth选项，那么-prune选项就会被find命令忽略
find -newer # 如果希望查找更改时间比某个文件新但比另一个文件旧的所有文件，可以使用-newer选项

grep ^gene  filename  #锚定行的开始 如：'^gene'匹配所有以gene开头的行。 
grep gene$    #锚定行的结束 如：'gene$'匹配所有以gene结尾的行。 
grep .    #匹配一个非换行符的字符 如：'gr.p'匹配gr后接一个任意字符，然后是p。  
grep *    #匹配零个或多个先前字符 如：'*grep'匹配所有一个或多个空格后紧跟grep的行。
grep .*   #一起用代表任意字符。  
grep []   #匹配一个指定范围内的字符，如'[Gg]rep'匹配Grep和grep。 
grep [^]  #匹配一个不在指定范围内的字符，如：'[^A-FH-Z]rep'匹配不包含A-R和T-Z的一个字母开头，紧跟rep的行。  
grep \(..\)    #标记匹配字符，如'\(love\)'，love被标记为1。   
grep \<        #锚定单词的开始，如:'\<grep'匹配包含以grep开头的单词的行。
grep \>        #锚定单词的结束，如'grep\>'匹配包含以grep结尾的单词的行。
grep x\{m\}    #重复字符x，m次，如：'0\{5\}'匹配包含5个0的行。 
grep x\{m,\}   #重复字符x,至少m次，如：'0\{5,\}'匹配至少有5个0的行。  
grep x\{m,n\}  #重复字符x，至少m次，不多于n次，如：'0\{5,10\}'匹配5--10个0的行。  
grep \w        #匹配文字和数字字符，也就是[A-Za-z0-9]，如：'G\w*p'匹配以G后跟零个或多个文字或数字字符，然后是p。  
grep \W        #\w的反置形式，匹配一个或多个非单词字符，如点号句号等。  
grep \b        #单词锁定符，如: '\bgrep\b'只匹配grep。


wc -c # 统计字节数
wc -l # 统计行数
wc -m # 统计字符数
wc -w # 统计词数，一个字被定义为由空白、跳格或换行字符分隔的字符串



