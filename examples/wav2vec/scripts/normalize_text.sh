python pre_filter.py $in_file temp.txt

python text_pre_process.py temp.txt temp2.txt --sent-end-marker=!

python text_nsw_process.py temp2.txt $out_file

rm -rf temp*.txt