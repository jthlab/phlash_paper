#!/bin/bash

chrom=$1
infile=$2
outfile=$3

L=$(head -n1 "$infile" | grep -oP '\-r \d+ \d+' | cut -f3 -d' ')


################################
# AWK scripts                  #
################################
read -r -d '' script << 'EOF'
BEGIN {
  last_pos = -1
}

NR == 1 {
  print "##fileformat=VCFv4.0"
  print "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">"
  printf "##contig=<ID=%s,length=%d>\n", chrom, L
  num_samples = (NF - 2) / 2
  printf "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT"
  header = ""
  for (i = 1; i <= num_samples; i++) {
    header = header "\tsample" i-1
  }
  print header
  next
}

{
  pos = int($1 + 0.5) # Round position to nearest integer
  if (pos == last_pos) { # Skip if this position was already printed
    next
  }
  last_pos = pos
  printf "1\t%s\t.\tA\tT\t.\t.\t.\tGT", pos
  for (i = 3; i <= NF; i+=2) {
    printf "\t%s|%s", $i, $(i+1)
  }
  printf "\n"
}
EOF
################################
# End of AWK Scripts           #
################################

tail -n+6 "$infile" | awk -v chrom=${chrom} -v L=${L} "$script" | bcftools view -o "$outfile"
