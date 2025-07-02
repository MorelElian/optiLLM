set -e
make main_dump
./main_dump llama3.2_1b.bin -i "a" -n 2 -s 42
cd /home/emorel/shared/clift/
make cliftdump
./cliftdump -m llama3.2_1b.bin -p "a" -n 2 -s 42
cd /home/emorel/work/optiLLM/
./compare_dumps.sh ./dump /home/emorel/shared/clift/dump phases.txt