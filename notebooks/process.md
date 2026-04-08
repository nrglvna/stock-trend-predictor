Quant Analysis: Before vs. After                                                                                                         
                                                    
  What Was Wrong and Why
                                                                                                                                           
  From a quant perspective the previous results had three compounding problems:
                                                                                                                                           
  1. Models were majority-class machines. With 53% Up days, every model discovered that predicting "Up" always gives 53% accuracy and zero 
  punishment from logloss on the minority class. Recall for Down was 0–26% across the board. A strategy built on this is long-only by      
  construction — it has no way to avoid down moves.                                                                                        
                                                                                                                                           
  2. LGBM was misconfigured for the data size. num_leaves=31, min_child_samples=20 on 860 training rows means leaves could contain as few
  as 20/860 = 2.3% of samples. The 5d model hit train accuracy 99.1% → test 53.0% — a classic memorization collapse. LGBM stopped at       
  iteration 1 on 1d/5d classification, meaning the regularization was so wrong it couldn't even make one useful tree.                    
                                                                                                                                           
  3. day_of_week was noise. Feature importance output from the notebook confirmed it ranked near the bottom. For a single stock over 5
  years you don't have enough Mondays (≈250) to learn a statistically reliable pattern. It was adding signal fragmentation to tree splits. 
                                                                                                                                         
  ---                                                                                                                                      
  Changes Made and Why Each One Helps
                                                                                                                                           
  ┌───────────────────────────────┬──────────────────────┬─────────────────────────────────────────────────────────────────────────────┐ 
  │            Change             │         File         │                               Quant Rationale                               │
  ├───────────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────┤   
  │ num_leaves 31→15,             │ config.py            │ 50/860 = 5.8% per leaf — matches recommended 5–10% rule for small datasets  │
  │ min_child_samples 20→50       │                      │                                                                             │   
  ├───────────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────┤   
  │ reg_alpha=0.1, reg_lambda=1.0 │ config.py            │ Explicit L1+L2 shrinkage; pushes weak-signal leaves to zero instead of      │
  │                               │                      │ overfitting                                                                 │   
  ├───────────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────┤   
  │ subsample/colsample 0.8→0.7   │ config.py            │ More aggressive stochastic subsampling = implicit regularization            │   
  ├───────────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────┤   
  │ class_weight='balanced'       │ classification.py    │ Forces equal loss weight per class. With 53/47 split, Down was being        │   
  │                               │                      │ penalized ~1.1× less — small but cumulative                                 │ 
  ├───────────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────┤   
  │ find_optimal_threshold()      │ classification.py    │ Default 0.5 threshold is wrong when class priors are unequal. Tuning on val │
  │                               │                      │  (applied to test) is legitimate                                            │   
  ├───────────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────┤ 
  │ Remove day_of_week            │ pipeline.py          │ Confirmed weak; removing it gives the model fewer spurious splits to        │   
  │                               │                      │ memorize                                                                    │   
  ├───────────────────────────────┼──────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │                               │ technical.py /       │ First feature using High/Low. Intraday range captures gap risk and          │   
  │ Add atr_ratio                 │ pipeline.py          │ sentiment differently from close-to-close vol. ATR spikes often precede     │   
  │                               │                      │ reversals — genuinely complementary to existing features                    │
  └───────────────────────────────┴──────────────────────┴─────────────────────────────────────────────────────────────────────────────┘   
                                                                                                                                           
  ---
  What to Expect After Re-running                                                                                                          
                                                                                                                                         
  Regression — no material change expected. R² near zero is the correct answer for daily close-to-close returns with technical features
  only. The signal simply isn't there at this horizon/frequency. Do not interpret near-zero R² as failure — it's the honest result.
                                          
  Classification — concrete expected improvements:
                                                                                                                                           
  class_weight='balanced': Down recall will rise from ~0–26% to ~35–50%. Precision for Down will fall proportionally. AUC is unaffected by
  threshold/weighting (it's rank-based) — so AUC is the cleanest measure of whether there's any signal at all.                             
                                                                                                                                         
  LGBM regularization: The 5d train/test collapse (99.1% → 53.0%) should shrink to something like 65–70% train → 55–58% test. LGBM will no 
  longer stop at iter=1 on 1d/5d.         
                                                                                                                                           
  Threshold tuning: For each model/horizon, call find_optimal_threshold(y_val, probas_val) to find the P(Up) threshold that maximizes      
  Down-class F1 on val, then evaluate test with that threshold. Expect F1-macro to improve 3–6 points over the fixed-0.5 baseline.
                                                                                                                                           
  ---                                                                                                                                      
  Honest Quant Assessment
                                                                                                                                           
  Is there any real edge? Logistic 1d AUC=0.628 (pre-tuning) is the single result worth examining. AUC of 0.55–0.65 is the realistic     
  ceiling for technical-only, single-stock, daily-frequency models. The academic literature consistently finds this range. It is a real but
   tiny edge.                             

  Would you trade this? Only under strict conditions:                                                                                      
  - Use AUC and Down-recall as primary metrics, not accuracy
  - Require consistent performance across multiple out-of-sample windows (walk-forward), not one static test set                           
  - Size positions based on P(Up) confidence, not binary class labels                                                                    
  - Treat it as one signal in a composite — not a standalone system  
                                                                                                                                           
  What still limits this model:               
  - The fundamental barrier is market efficiency at daily frequency. Close-to-close returns embed everything already known                 
  - 860 training rows is genuinely small — walk-forward with 3-year rolling train would give you ~5 test windows instead of 1, which is the
   minimum needed to trust any metric                                                                                                      
  - Single stock, single asset class — no cross-sectional signal, no regime detection, no macro context                                    
                                                                                                                                           
  What would move the needle most: Adding cross-sectional features (AAPL return relative to SPY, VIX level, sector momentum) would likely
  double the AUC improvement compared to any further tuning of these 11 features.   