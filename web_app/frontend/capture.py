import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        # Set a wide, professional viewport
        await page.set_viewport_size({"width": 1400, "height": 900})
        
        print("Navigating to CDSS interface...")
        await page.goto("http://localhost:8080/index.html")
        
        print("Switching language to English...")
        # change the dropdown
        await page.select_option("#lang-switch", "en")
        # trigger the change event
        await page.evaluate("switchLanguage('en')")
        await page.wait_for_timeout(500)
        
        print("Loading random patient data...")
        await page.evaluate("loadRandomPatient()")
        await page.wait_for_timeout(500)
        
        print("Running causal risk assessment...")
        await page.evaluate("runAssessment()")
        # Wait for the dashboard to render (the function has a 800ms timeout)
        await page.wait_for_timeout(2000)
        
        print("Taking screenshot...")
        await page.screenshot(path="C:/Dynamic-RRT/figures/Fig12_CDSS_Interface.png")
        print("Done!")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
