#!/bin/bash

# submit with
# condor-run run_convert.sh -a inputs.list -e <prod> <converter> -x ~/x509up_u51268

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
  CLASSES="electron_pion"
  FLAG="-F"
else
  CLASSES="all"
  FLAG=""
fi

export X509_USER_PROXY=$PWD/x509up_u51268

xrdcp root://eoscms.cern.ch/$(echo $INPUT | sed 's|/eos/cms||') $PWD

$SMALLCALO/bin/convert.py $CONVERTER $PWD/$(basename $INPUT) out.tfrecords $NEVENTS $FLAG

xrdcp out.tfrecords root://eoscms.cern.ch//store/user/yiiyama/small_calorimeter/$CLASSES/$PROD/$CONVERTER/$(basename $INPUT | sed 's/[.]root/.tfrecords/')
