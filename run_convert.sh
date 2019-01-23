#!/bin/bash

# submit with
# condor-run run_convert.sh -i inputs.list -e "<prod> <converter> <class>" -x ~/x509up_u51268

PROD=$1
CONVERTER=$2
CLS=$3
INPUT=$4
NEVENTS=$5

cd $HOME/cmssw/CMSSW_10_2_5_SL6/
eval `scram runtime -sh`
cd -

SMALLCALO=$HOME/src/small_calorimeter

export PYTHONPATH=$SMALLCALO/lib:$PYTHONPATH

if [ $CLS = "epi" ]
then
  CLASS="electron_pion"
  FLAG="-F"
  ARG=""
elif [ $CLS = "gpi" ]
then
  CLASS="gamma_pion"
  FLAG="-G"
  ARG=""
elif [ $CLS = "gpir" ]
then
  CLASS="gamma_pion_reverse"
  FLAG="-G"
  ARG="-R"
elif [ $CLS = "easy" ]
then
  CLASS="gamma_pion_easy"
  FLAG="-F"
  ARG='-f "true_energy < 10 && true_x > 0 && true_y > 0"'
else
  CLASS="all"
  FLAG=""
  ARG=""
fi

export X509_USER_PROXY=$PWD/x509up_u51268

xrdcp root://eoscms.cern.ch/$(echo $INPUT | sed 's|/eos/cms||') $PWD

source $SMALLCALO/setup.sh

$SMALLCALO/bin/convert.py $FLAG $ARG $CONVERTER $PWD/$(basename $INPUT) out.tfrecords $NEVENTS

CONVERTER=$(echo $CONVERTER | sed 's/.*[.]\(.*\)/\1/')

xrdcp out.tfrecords root://eoscms.cern.ch//store/user/yiiyama/small_calorimeter/$CLASS/$PROD/$CONVERTER/$(basename $INPUT | sed 's/[.]root/.tfrecords/')
