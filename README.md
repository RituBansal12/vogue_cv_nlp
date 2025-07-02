# vogue_cv_nlp
Analyzing vogue covers from June 1935 to June 2025 to identify key trends in fashion. 

Data Extraction:
Extracted data from Vogue Archive using proquest through Toronto Public Library. Downloaded PDF of cover images and metadata between the June 1935-June 2025. 
1. extract_covers.py - Goes over the raw pdf to extract cover images and link it to the publish date using metadata
2. extract_metadata.py - Extracts metadata about the cover such as Volume, Issue, Editor, Photographer, Caption, etc., 

Data Processing:
1. extract_cover_attributes.py - Using FashionCLIP for no shot item recognition for cover images to identify attributes about the apparel 
2. extract_metadata_attributes.py - Using NER(hugging face transformers), KeyBert(Keyphrase extraction), and SpaCy(Noun Phrase extraction) to identify brands, people, and apparel attributes mentioned in the caption.