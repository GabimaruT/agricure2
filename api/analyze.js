export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).end();

  const ML_URL = process.env.ML_SERVER_URL ||
    'https://YOUR-RAILWAY-URL.up.railway.app';

  try {
    const response = await fetch(`${ML_URL}/api/predict-image`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body),
      signal: AbortSignal.timeout(25000)
    });

    if (!response.ok) throw new Error('ML server error');
    const data = await response.json();
    return res.status(200).json(data);

  } catch (error) {
    return res.status(500).json({
      success: false,
      error: error.message,
      isPlant: false,
      disease: 'Server Error',
      severity: 0,
      treatments: ['Please try again later']
    });
  }
}
