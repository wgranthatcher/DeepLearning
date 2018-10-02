#!/bin/bash
#takes a directory as an argument
#will create final fomratted text file in target DecisionTreeClassifier

target=${1%/}
rm $target/{perm_errors.txt,perm1.txt,perm2.txt,perm_final.txt} 2> /dev/null

for file in $target/*.apk; do
    aapt d permissions "$file" 2>> $target/perm_errors.txt 1>> $target/perm1.txt
    tr '\n' ' ' < $target/perm1.txt > $target/perm2.txt
    cat $target/perm2.txt | sed 's/package:/\n/g' > $target/perm_final.txt
done;

echo "cleaning up"
rm $target/{perm1.txt,perm2.txt}

n=($(wc -l $target/perm_errors.txt))
m=($(wc -l $target/perm_final.txt))
echo "$n apks failed to extract"
echo "$m apks extracted succesfully"
