#!/usr/bin/env bash

virtualenv virtual
deactivate
source virtual/bin/activate
pip install -t lib -r requirements.txt