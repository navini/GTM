#!/bin/bash

EXEC_LOC="/s/chopin/a/grad/navinid/Dropbox/Research/gtm/python_GTM/GTMClassification/"
EXEC_FILE="$EXEC_LOC"run2.sh
machinename="machines"
filenames=( `ls -l ./Data | grep ".txt" | awk '{print $9}'` )
LOG_FILE="Njobshop.sh.log"


l=""

#List machine names from machines file
while read -r line
do
  IFS=' ' read -a array <<< "$line"
#  echo "${array[0]}" >> $machine_names
#  echo "ssh ${array[0]} w >> $machine_state"
  l="${l} ${array[0]}"
  #ssh -T ${array[0]} w >> $machine_state
  #echo "`hostname` end" 
done < "$machinename"

echo "l=$l"
echo "start-------------------------------------------------------------------------------------------------------------------------------------" >> ${LOG_FILE}   


id=0
for i in $l; 
do 
#  echo "$i"  
#  printf "%s %s\n" "$i" "${l2[1]}"  
  echo "$i ${filenames[$id]}" >> ${LOG_FILE}
#  INPUT_FILE=$EXEC_LOC$INPUT_FILE_PREFIX$id
#  echo "$i :  $EXEC_FILE $INPUT_FILE 0 $EXEC_LOC" >> "jobshopScript.sh.log"   
  #ssh $i "nohup sh $EXEC_FILE $INPUT_FILE 0 $EXEC_LOC &" &
  nohup ssh $i "nohup sh $EXEC_FILE ${filenames[$id]} ${EXEC_LOC} &> ${EXEC_LOC}${i}"_my_2.out" &" &>> `hostname`"_my.out" &
  ((id=$id+1))
done

echo "stop-------------------------------------------------------------------------------------------------------------------------------------" >> ${LOG_FILE}   
