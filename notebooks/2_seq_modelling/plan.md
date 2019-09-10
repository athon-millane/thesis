# Unsupervised Training
- Upscale to biggest CPU machine, and train + save all the SentencePiece models
- Train with 100,000 rows minimum
- Ideally want to churn through the entire reference genome

# Paper Writing
- Write up first experiment - 10 pages
    - Go through everything I explored
    - Add figures that are interesting
    - Talk about data
    - Discuss, interpretation is important
    - Discuss computational working environment
    - Details on tools and preference
- Discussion of difference between distributed and Sequential representations

# Supervised training
- Ensure data is up to scratch - focus on promotors
- Run working example and build from there
- Plot ROC, Accuracy, etc

# Unsupervised Promotor Neuron
- find neuron with greatest variance across promotorID problem.
- visualise change in neuron weights across tokens.
- show unsupervised learning conditions neurons to recognise promotor regions
  - as per https://openai.com/blog/unsupervised-sentiment-neuron/