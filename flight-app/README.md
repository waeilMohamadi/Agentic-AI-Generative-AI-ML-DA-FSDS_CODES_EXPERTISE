# Halcyon — Flight Search App

A React flight search and booking demo. Search sample routes across major airports, filter by stops and sort order, and complete a local reservation (stored in the browser).

## Features

- Full-bleed search home with airport, date, cabin, and traveler selection
- Results page with price / duration / departure sorting and stop filters
- Booking flow with trip summary and confirmation code
- Sample inventory for domestic and international city pairs (no live airline API)

## Stack

- React 19 + TypeScript
- Vite
- React Router

## Run locally

```bash
cd flight-app
npm install
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`).

```bash
npm run build
npm run preview
```

## Sample routes that return flights

- JFK → LAX
- SFO → SEA
- BOS → LHR
- LAX → NRT
- ORD → CDG

## Notes

Prices and schedules are illustrative. Confirmations are saved to `localStorage` under `halcyon-last-booking` and are not sent to an airline or payment provider.
