#! /bin/bash
cat requirements.txt|cut -d'=' -f1|xargs pip install
