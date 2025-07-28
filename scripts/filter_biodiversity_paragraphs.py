"""
Biodiversity Paragraph Filter for BiodiversityASSET Project

This script filters paragraphs for biodiversity relevance using:
1. ESGBERT/EnvironmentalBERT-biodiversity transformer model
2. Hard-coded biodiversity keyword matching

Input: CSV files from data/processed/extracted_paragraphs_from_pdfs/
Output: CSV files in data/processed/biodiversity_related_paragraphs/
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import pandas as pd
import re
import os
import unicodedata
from pathlib import Path
from tqdm import tqdm

# Define project directories using relative paths
PROJECT_ROOT = Path(__file__).parent.parent  # Go up from scripts/ to project root
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "extracted_paragraphs_from_pdfs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "biodiversity_related_paragraphs"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Biodiversity transformer model configuration
TOKENIZER_NAME = "ESGBERT/EnvironmentalBERT-biodiversity"
MODEL_NAME = "ESGBERT/EnvironmentalBERT-biodiversity"

# Flat list of biodiversity keywords
bio_keywords = [ 
    "biodiversity", "biodiversity conservation", "biodiversity protection", "biodiversity management",
    "Marine", "Marine shelf", "Pelagic ocean waters", "Deep sea floors", "Anthropogenic marine", 
    "Bioherm Biogeome", "Hydrothermal Biogeome",
    "Freshwater", "Rivers and streams", "Lakes", "Artificial wetlands", 
    "Rheobiogeome", "Limnobiogeome",
    "Subterranean", "Anthropogenic subterranean voids", 
    "Terrestrial", "Tropical or subtropical forests", "Temperate-boreal forests and woodlands", 
    "Shrublands and shrubby woodlands", "Savannas and grasslands", "Deserts and semi-deserts", 
    "Polar or alpine (cryogenic)", "Intensive land use", 
    "algae", "organisms", "coral reefs", "crustaceans", "dolphins", "marine fungi", "marine invertebrates",
    "mollusks", "plankton", "seagrass conservation", "seaweed", "sponges", "stormwater retention basins",
    "turtles", "whales", "habitat protection", "nature conservation projects", "rare species protection",
    "natural habitat restoration", "invasive species control", "amphibian habitats", "beetle biodiversity",
    "honeybee conservation", "butterfly gardens", "coastal shelf", "latitudinal zonality", "upwelling",
    "sediments", "planktonic algae", "macrophytes", "nekton", "benthos", "periphyton", "albatross",
    "clean rivers", "drifting algae", "ocean", "ocean resources", "open-ocean predators", "pelagic fish",
    "phytoplankton", "squid", "swordfish", "tuna", "whale sharks", "zooplankton", "photic zone",
    "aphotic zone", "downwelling", "detrital food webs", "pasture food chains", "bioluminescent species",
    "coastal ecosystems", "deep-sea corals", "extremophiles", "giant isopods", "glass sponges", "hagfish",
    "sea cucumbers", "terrestrial ecosystems", "tube worms", "vent crabs", "abyssal zone",
    "sediment accumulation", "ephaptonic forms", "burrowing fauna", "low biodiversity", "fish farms",
    "introduced species", "invasive marine species", "marine fauna", "marine habitats", "marine indicators",
    "marine mammals", "pollution impacts on marine life", "sustainable aquaculture", "high water temperature",
    "waves", "tides", "hermatypic corals", "trophosymbiosis", "endosymbionts", "light penetration",
    "deep sea", "hydrothermal vents", "mineral-rich fluids", "endosymbiotic chemosynthesis",
    "burrowing organisms", "amphibians", "aquatic insects", "dragonflies", "floodplain vegetation",
    "freshwater fish", "freshwater mussels", "hydrological regulation species", "macroinvertebrates",
    "native woodland species", "otters", "re-meandered streams", "riparian", "riparian buffers",
    "riparian forest buffer", "riparian vegetation", "river", "stream algae", "streams",
    "understory biodiversity", "wildlife corridors", "land use", "savanna conservation", "tiger reserves",
    "pollinator habitats", "wildflower meadows", "ammonia-tolerant invertebrates", "aquatic plants",
    "denitrifying bacteria", "duckweed", "freshwater sponge", "frog habitats", "lake fish", "macroalgae",
    "migratory birds", "nitrogen-sensitive aquatic species", "reed beds", "snails", "submerged macrophytes",
    "water beetles", "waterfowl", "wetland plants", "freshwater lakes", "reservoirs", "stratification",
    "amphibian refuge", "beneficial insects", "constructed wetland biota", "cover crops",
    "crop pollinators", "freshwater", "freshwater biodiversity", "freshwater resources", "habitat diversity",
    "invertebrates in ditches", "lakes", "organic farming", "peatland", "pollination networks",
    "reed-dwelling species", "snail hosts", "soil microfauna", "soil organisms", "vegetation cover",
    "water purification species", "wetland", "wetland birds", "wetland flora", "oxidizing conditions",
    "substrate diversity", "detrital zones", "rainforest canopy species", "orchids", "tropical birds",
    "jaguars", "leafcutter ants", "tropical frogs", "parrots", "vines", "understory mammals", "biodiversity",
    "species", "forest conservation", "reforestation", "afforestation", "forest biodiversity",
    "flora regeneration", "fauna conservation", "tree planting", "native species planting",
    "forest habitat corridors", "sustainable forestry", "big trees", "trophic specialization",
    "complex community structure", "plant richness", "invertebrates", "flora", "fauna", "coniferous trees",
    "boreal mosses", "temperate deer", "woodpeckers", "lynx", "boreal owls", "spruce forest fauna",
    "fungal diversity", "forest floor organisms", "conifers", "trees", "fungi", "insects",
    "soil invertebrates", "phytophagous insects", "insectivorous birds", "mediterranean shrubs",
    "heathland birds", "reptiles", "scrubland mammals", "flowering shrubs", "pollinator species",
    "small ground-nesting birds", "dryland butterflies", "ground beetles", "herbaceous diversity",
    "high nature value farmland", "overgrazing-sensitive species", "species-rich grassland",
    "steppe birds", "steppe", "soil", "humic soils", "plants", "large herbivores", "grassland",
    "succulents", "cacti", "desert foxes", "camels", "sand lizards", "scorpions", "desert beetles",
    "nocturnal pollinators", "adapted grasses", "arid climate", "sandy soil", "succulent plants",
    "burrowing animals", "low vegetation", "ephemeral species", "lichen", "alpine cushion plants",
    "snow leopards", "arctic foxes", "moss tundra", "caribou", "penguins", "ice algae",
    "short growing season", "peat formation", "shrubs", "grasses", "small rodents", "urban birds",
    "crop-adapted pollinators", "hedgerow species", "soil nematodes", "domestic biodiversity",
    "garden invertebrates", "urban-adapted mammals", "weeds", "feral species", "sustainable agriculture",
    "regenerative agriculture", "horticulture", "urban farming", "agroforestry", "permaculture",
    "wildlife-friendly farming", "ecological garden maintenance", "organic gardening",
    "low-impact food systems", "responsible plant cultivation", "biodegradable materials",
    "large community gardens", "agro-biodiversity", "urban forests",
    "soy beans", "palm oils", "cattle", "beef", "air pollution", "food chain", "fisheries",
    "prawns", "octopus", "mangroves"
]

def preprocess_paragraph(text):
    """
    Preprocess paragraph to normalize spacing, dashes, and characters for exact matching.
    """
    
    text = unicodedata.normalize("NFKD", text.lower())
    text = re.sub(r"[-_]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def contains_keyword(text, keywords):
    """
    Check if the preprocessed text contains any keyword from the list.
    Returns True if at least one keyword is found.
    """
    cleaned_text = preprocess_paragraph(text)
    for keyword in keywords:
        cleaned_keyword = preprocess_paragraph(keyword)
        pattern = r"\b" + re.escape(cleaned_keyword) + r"\b"
        if re.search(pattern, cleaned_text):
            return True
    return False

def classify_with_transformer_and_keywords(paragraphs, pipe, keywords, file_name_column, csv_file_name, threshold=0.8):
    """
    Classify paragraphs using ESGBERT/EnvironmentalBERT-biodiversity and enhance with keyword matching.

    :param paragraphs: List of paragraphs to classify.
    :param pipe: Biodiversity transformer pipeline.
    :param keywords: List of strings, where each string is a biodiversity-related keyword or phrase.
                     Each paragraph is checked individually for the presence of these keywords using
                     exact match after normalization (e.g., lowercase, dash removal, punctuation cleanup).
    :param file_name_column: List of file names corresponding to each paragraph.
    :param csv_file_name: Name of the CSV file being processed (used for tracking).
    :param threshold: Confidence threshold for including transformer-predicted paragraphs.

    :return: A list of filtered paragraph dictionaries containing matched paragraphs and metadata.
    """
    results = []
    seen = set()
    
    # Add progress bar for processing paragraphs
    with tqdm(total=len(paragraphs), desc="Processing paragraphs", unit="para") as pbar:
        for i, (paragraph, file_name) in enumerate(zip(paragraphs, file_name_column)):
            # Try transformer classification if model is available
            if pipe is not None:
                try:
                    prediction = pipe(paragraph, padding=True, truncation=True)
                    confidence = prediction[0]["score"]
                    label = prediction[0]["label"]
                except Exception as e:
                    print(f"Error processing paragraph {i + 1} with transformer: {e}")
                    confidence = 0
                    label = "LABEL_0"
            else:
                # Skip transformer if model not available
                confidence = 0
                label = "LABEL_0"

            keyword_matched = contains_keyword(paragraph, keywords)
            transformer_matched = (label == "biodiversity" and confidence > threshold)

            if transformer_matched or keyword_matched:
                entry = {
                    "pdf_file_name": file_name,
                    "csv_file_name": csv_file_name,
                    "paragraph_text": paragraph,
                    "matched_by_keyword": True if keyword_matched else False,
                    "matched_by_transformer": True if transformer_matched else False,
                }
                if tuple(entry.items()) not in seen:
                    results.append(entry)
                    seen.add(tuple(entry.items()))
                    
                # Update progress bar description with current matches
                pbar.set_postfix({"matches": len(results)})
            
            # Update progress bar
            pbar.update(1)

    return results

def save_filtered_paragraphs(filtered_paragraphs, output_path):
    """
    Save filtered paragraphs to a CSV file.
    """
    try:
        pd.DataFrame(filtered_paragraphs).to_csv(output_path, index=False, encoding='utf-8')
        print(f"Filtered paragraphs saved to: {output_path}")
    except Exception as e:
        print(f"Error saving filtered paragraphs to {output_path}: {e}")

def main():
    """Main function to run the biodiversity paragraph filtering process"""
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    if not INPUT_DIR.exists():
        print(f"Error: Input directory {INPUT_DIR} does not exist!")
        print("Please run the extract_pdfs.py script first to generate input data.")
        return
    
    # Get all CSV files from the input directory
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".csv")]
    
    if not csv_files:
        print(f"No CSV files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")
    print("Loading biodiversity transformer model...")
    
    # Load the model here to avoid reloading for each file
    try:
        print("This may take a few minutes on first run...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, max_len=512)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        print("✓ Transformer model loaded successfully")
    except Exception as e:
        print(f"Error loading transformer model: {e}")
        print("Continuing with keyword-only filtering...")
        pipe = None
    
    total_filtered = 0
    
    # Add progress bar for processing files
    for file in tqdm(csv_files, desc="Processing CSV files", unit="file"):
        file_path = INPUT_DIR / file
        print(f"\nProcessing file: {file}")
        
        try:
            # Read CSV with the expected column names from extract_pdfs.py
            data = pd.read_csv(file_path)
            
            # Validate required columns exist
            required_columns = ["pdf_file_name", "paragraph_id", "paragraph_text"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                print(f"Error: Missing columns {missing_columns} in {file}")
                continue

            paragraphs = data["paragraph_text"].dropna().tolist()
            file_names = data["pdf_file_name"].dropna().tolist()
            
            if len(paragraphs) == 0:
                print(f"No paragraphs found in {file}")
                continue
                
            print(f"Processing {len(paragraphs)} paragraphs...")

            filtered_paragraphs = classify_with_transformer_and_keywords(
                paragraphs, pipe, bio_keywords, file_names, file, threshold=0.8
            )

            if filtered_paragraphs:
                # Create output filename
                output_file = OUTPUT_DIR / f"{file}"
                save_filtered_paragraphs(filtered_paragraphs, output_file)
                total_filtered += len(filtered_paragraphs)
                print(f"Found {len(filtered_paragraphs)} biodiversity-related paragraphs")
            else:
                print(f"No biodiversity-related paragraphs found in {file}")
                
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    print(f"\nProcessing complete!")
    print(f"Total filtered paragraphs across all files: {total_filtered}")
    print(f"Output files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
