const { test, expect } = require('@playwright/test');
const path = require('path');

test('upload sample CSV and show results', async ({ page }) => {
  await page.goto('/');

  // Ensure upload form is present
  const fileInput = page.locator('input[type=file]')
  await expect(fileInput).toBeVisible()

  // Upload fixture CSV
  const fixture = path.join(__dirname, '..', 'test-fixtures', 'sample_features.csv')
  await fileInput.setInputFiles(fixture)

  // Submit form
  await page.locator('button:has-text("Upload & Predict")').click()

  // Wait for Prediction header
  await expect(page.locator('text=Prediction')).toBeVisible({ timeout: 120000 })

  // Expect probabilities chart canvas
  await expect(page.locator('canvas')).toBeVisible()

  // Expect either a table or top features cards / raw JSON
  await expect(page.locator('text=Top features')).toBeVisible()

  // Check that predicted class is displayed
  await expect(page.locator('text=Predicted class')).toBeVisible()
});
