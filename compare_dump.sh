set -e
make forward_dump
./forward_dump llama3.2_1b.bin -i "a" -n 1
cd /home/emorel/shared/clift/
make cliftdump
./cliftdump -m llama3.2_1b.bin -p "a" -n 1
cd /home/emorel/work/optiLLM/
compare_dumps -f phases.txt /home/emorel/shared/clift/dump ./dump