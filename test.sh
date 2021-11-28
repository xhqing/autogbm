strA="helloworld"
strB="autogbm"
if [[ PS1 =~ $strB ]]
then
    echo "包含"
else
    echo "不包含"
fi
