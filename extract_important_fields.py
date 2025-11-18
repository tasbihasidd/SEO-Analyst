import json

# IMPORTANT KEYS WE WANT TO KEEP
IMPORTANT_FIELDS = {
    "url",
    "title",
    "description",
    "word_count",
    "keywords",
    "bigrams",
    "trigrams",
    "warnings",
    "headings",
    "additional_info"
}

IMPORTANT_ADDITIONAL = {
    "title",
    "canonical",
    "og_title",
    "og_desc"
}

def extract_important_seo_fields(input_json_path: str, output_json_path: str):
    """
    Extract only important SEO fields from full SEO report JSON.
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    optimized_pages = []

    for page in data.get("pages", []):
        optimized_page = {}

        for key, value in page.items():

            # Only save important top-level fields
            if key in IMPORTANT_FIELDS:

                # Additional info needs filtering
                if key == "additional_info":
                    filtered_additional = {
                        k: v for k, v in value.items() if k in IMPORTANT_ADDITIONAL
                    }
                    optimized_page[key] = filtered_additional
                else:
                    optimized_page[key] = value
        
        optimized_pages.append(optimized_page)

    # Final compact JSON
    final_data = {"pages": optimized_pages}

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

    print(f"Optimized SEO JSON saved to: {output_json_path}")


# Run directly
if __name__ == "__main__":
    extract_important_seo_fields(
        input_json_path="seo_report.json",
        output_json_path="seo_report_optimized.json"
    )