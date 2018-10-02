#/bin/sh
#takes target directory as argument
#will create final edited text file in targe dir

target=${1%/}
rm $target/feature_errors.txt $target/feature1.txt $target/feature2.txt $target/feature_final.txt 2> /dev/null

for file in $target/*.apk; do aapt dump badging "$file" 2>> $target/feature_errors.txt | grep 'feature' >> $target/feature1.txt ; done;
tr '\n' ' ' < $target/feature1.txt > $target/feature2.txt;
cat $target/feature2.txt | sed 's/feature-group:/\nfeature-group:/g' > $target/feature_final.txt;

echo "cleaning up..."
rm $target/{feature1.txt,feature2.txt}

n=($(wc -l $target/feature_errors.txt))
m=($(wc -l $target/feature_final.txt))
echo "$n apks failed to extract"
echo "$m apks extracted succesfully"
