#/bin/sh
#takes target directory as argument
#will create final edited text file in targe dir

target=${1%/}
rm $target/badging_errors.txt $target/badging1.txt $target/badging2.txt $target/badging_final.txt 2> /dev/null

for file in $target/*.apk; do
    echo $(basename "$file") >> $target/badging1.txt
    aapt dump badging "$file" 2>> $target/badging_errors.txt | grep -e "name='[^']*'" -o | sort -u >> $target/badging1.txt ; done;
tr '\n' ' ' < $target/badging1.txt > $target/badging2.txt;
cat $target/badging2.txt | sed 's/\ [^\ ]*\.apk/\n&/g' > $target/badging_final.txt;

echo "cleaning up..."
rm $target/{badging1.txt,badging2.txt}

n=($(wc -l $target/badging_errors.txt))
m=($(wc -l $target/badging_final.txt))
echo "$n apks failed to extract"
echo "$m apks extracted succesfully"
