#!/bin/bash

apt-get update
apt-get upgrade
apt install wget
apt install curl
apt install git

python main.py -task install