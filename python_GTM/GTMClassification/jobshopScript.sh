#!/bin/bash

EXEC_LOC="/s/chopin/l/grad/waruna/Dropbox/CSU/TA/Spring-2014/CS575/HW3/grading/"
EXEC_FILE="$EXEC_LOC"run.sh
filename="machines"
machine_names="machine.names_tmp"


rm $machine_names

l=""
#List machine names from machines file
while read -r line
do
  IFS=' ' read -a array <<< "$line"
#  echo "${array[0]}" >> $machine_names
#  echo "ssh ${array[0]} w >> $machine_state"
  l="$l ${array[0]}"
  #ssh -T ${array[0]} w >> $machine_state
  #echo "`hostname` end" 
done < "$filename"

l2=""

grade=HW3

cd $grade
id=0
for entry in "."/*
do
#  echo "$entry"
  if [ -f $entry ]
  then
    #
#    echo "file $entry"
    echo ""
  else
    l2[$id]="$entry"
    ((id=$id+1))
  fi
done


echo "l=$l"
echo "l2=${l2[*]}"
echo "start-------------------------------------------------------------------------------------------------------------------------------------" >> "jobshopScript.sh.log"   

id=0

for i in $l; 
do 
#  echo "$i"  
#  printf "%s %s\n" "$i" "${l2[1]}"  
  echo "$i ${l2[$id]}"
#  INPUT_FILE=$EXEC_LOC$INPUT_FILE_PREFIX$id
#  echo "$i :  $EXEC_FILE $INPUT_FILE 0 $EXEC_LOC" >> "jobshopScript.sh.log"   
  #ssh $i "nohup sh $EXEC_FILE $INPUT_FILE 0 $EXEC_LOC &" &
  nohup ssh $i "nohup sh $EXEC_FILE ${l2[$id]} $EXEC_LOC &" &

  ((id=$id+1))
done

echo "stop-------------------------------------------------------------------------------------------------------------------------------------" >> "jobshopScript.sh.log"   
