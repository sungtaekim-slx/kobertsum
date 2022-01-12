#!/bin/bash

apt-get -y update
apt-get -y upgrade
apt install -y wget
apt install -y curl
apt install -y git

python main.py -task install
