# Northeastern MATH4581 Final Project Problem Fall 2025

## Problem
2. We have looked at a short version of tennis (where they start at deuce). Do the full version: set up the
transition matrix for a game and find the probability that A wins the game (do games where A serves and
B serves, choose an actual player and find the probability that they win a point when they serve and when
the other player serves); set up the transition matrix for a set where A and B alternate serving and find the
probability that A wins. Finally, assume they are playing a best-of-three sets match. Find the probability that A wins.

## Details
I implemented a general solver for this problem with a few simplifications:
* The specified player always serves first for each set
* Sets have no tie break and are win by 2 with 5-5 as the tied/deuece-like state, e.g a lost game at 6-5 --> 5-5 (tied)

The program automatically runs an example for Aryna Sabalenka, but you can provide either an ATP or a WTA player's stat URL 
to generate tables and percentages for their 2025 stats. You can also just manually input values for serve/return percentages

## Required Packages
numpy
