# Stage 1: Build the Next.js app
FROM node:20.18.0-alpine AS builder

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json ./
RUN npm install --legacy-peer-deps

# Copy the source code and build the app
COPY . .
RUN npm run build

# Stage 2: Run the production app
FROM node:20.18.0-alpine AS runner

WORKDIR /app

# Copy the standalone build and public assets from the builder stage
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

# Expose the application port
EXPOSE 3000

# Start the Next.js app in production mode
CMD ["node", "server.js"]
