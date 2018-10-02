#!/bin/bash

#the purpose of this script is to iterate through apk repository directories
#and mark all .apk's where the manifest cannot be extracted. 
#these files will be left in place and .broken appended to them.

target=${1%/}
i=0
for file in $target/*.apk; do
	aapt dump badging "$file" &> /dev/null;
	if test $? -eq 1; 
		then
			mv "$file" "$file.broken";
			$i = $i + 1
	fi
done;
t=$(ls -l $target | wc -l)
echo $i/$t files found to be broken

