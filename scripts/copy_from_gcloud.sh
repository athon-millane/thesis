#!/bin/bash

gcloud auth login
gcloud auth application-default login

gsutil cp -r gs://athon-research/genomeXL/models .