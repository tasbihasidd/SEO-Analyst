import json
from pyseoanalyzer import analyze

def generate_seo_report(url: str, output_json_path: str):
    """
    SEO report generate karo aur usko JSON file mein save karo.
    """

    report = analyze(
        url,
        follow_links=True,
        analyze_headings=True,
        analyze_extra_tags=True
    )

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    print(f"SEO report saved to: {output_json_path}")


if __name__ == "__main__":
    website_url = "https://airobotixs.com/"
    output_file = "seo_report.json"

    generate_seo_report(website_url, output_file)