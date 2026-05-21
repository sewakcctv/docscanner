# Doc Scanner

A minimal document scanner web app built for mobile. Open it in your phone's browser, point the camera at a document, capture it, and save it to your gallery.

## Features

- Uses the rear camera for best document quality
- Capture with one tap
- Save via the system share sheet (saves to gallery on Android) or fallback download
- Works offline after first load (no backend needed)

## Run Locally

**Prerequisites:** Node.js

```bash
npm install
npm run dev
```

Then open `http://localhost:3000` on your phone (must be on the same Wi-Fi network, or use a tunnel like `ngrok`).

## Build for Production

```bash
npm run build
```

Deploy the `dist/` folder to any static hosting (GitHub Pages, Netlify, Vercel, etc.).

## Notes

- Camera permission must be granted in the browser
- On Android Chrome, the Web Share API saves directly to the gallery
- On browsers without share support, a download is triggered instead (saves to Downloads folder)
