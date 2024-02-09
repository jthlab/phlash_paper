#!/bin/sh

file=$1
k=$2  # number of samples

bcftools view -H $file | awk -v k="$k" 'BEGIN {FS="\t"; OFS="\t"; prev_pos=0} 
    !/^#/ { 
        chrom=$1; pos=$2; genotypes=""; 
        for(i=10; i<10+k && i<=NF; i++) { 
            split($i, arr, /[|\/]/); 
            genotypes=genotypes arr[1] arr[2]; 
        } 
        if(NR==1) { 
            print chrom, pos, pos, genotypes; 
            prev_pos=pos; 
        } else { 
	    diff = pos-prev_pos;
            if(diff != 0) {  # Check if the position difference is not zero
                print chrom, pos, diff, genotypes;
                prev_pos=pos;
            }
        } 
    }'
