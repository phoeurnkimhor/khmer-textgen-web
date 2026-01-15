# Use Node 20 on Debian slim (for glibc compatibility with lightningcss)
FROM node:20-slim AS base

# Set working directory
WORKDIR /app

# Install dependencies only when needed
FROM base AS deps
COPY package.json package-lock.json* ./
RUN npm ci

# Build the application
FROM base AS builder
WORKDIR /app

# Install build tools for native dependencies
RUN apt-get update && apt-get install -y build-essential python3 rustc cargo && rm -rf /var/lib/apt/lists/*

# Copy app source and dependencies
COPY . .
COPY --from=deps /app/node_modules ./node_modules

# Rebuild native modules to match container environment
RUN npm rebuild lightningcss

# Build the Next.js app
RUN npm run build

# Production image
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000

# Copy only necessary output files
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

# Expose port 3000
EXPOSE 3000

# Start the Next.js server
CMD ["npm", "start"]
