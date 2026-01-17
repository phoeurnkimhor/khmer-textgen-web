# Use Node 20 on Debian slim (for glibc compatibility with lightningcss)
FROM node:20-bookworm AS base

# Install pnpm globally
RUN npm install -g pnpm

# Set working directory
WORKDIR /app    

# Install dependencies only when needed
FROM base AS deps
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

# Build the application
FROM base AS builder
WORKDIR /app

# Build argument for API URL
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

# Install build tools for native dependencies
RUN apt-get update && apt-get install -y build-essential python3 rustc cargo && rm -rf /var/lib/apt/lists/*

# Copy app source and dependencies
COPY . .
COPY --from=deps /app/node_modules ./node_modules

# Rebuild native modules to match container environment
RUN pnpm rebuild lightningcss

RUN pnpm run build

# Production image
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production

# Copy exported static files (Next.js exports to 'out' directory)
COPY --from=builder /app/out ./out

# Simple HTTP server to serve static files
RUN npm install -g serve
CMD ["serve", "-s", "out", "-l", "3300"]

# Expose port 3300
EXPOSE 3300