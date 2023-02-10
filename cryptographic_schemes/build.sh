#!/bin/bash

apt-get install build-essential cmake git libgmp3-dev libprocps-dev python-markdown libboost-all-dev libssl-dev -y

add-apt-repository ppa:webupd8team/java

apt-get update -y

apt-get install oracle-java8-installer

apt-get install junit4

wget https://www.bouncycastle.org/download/bcprov-jdk15on-159.jar

git clone --recursive https://github.com/akosba/jsnark.git

cd jsnark/libsnark


git submodule init && git submodule update

mkdir build && cd build && cmake .. && make

cd ../..

cp -r ../ourProject/gadgets/IOT JsnarkCircuitBuilder/src/examples/gadgets
cp -r ../ourProject/generators/IOT JsnarkCircuitBuilder/src/examples/generators

cd JsnarkCircuitBuilder

mkdir -p bin

javac -d bin -cp /usr/share/java/junit4.jar:../../bcprov-jdk15on-159.jar  $(find ./src/* | grep ".java$")
