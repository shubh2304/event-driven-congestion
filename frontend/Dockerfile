# Stage 1: Build the Next.js application
FROM node:18-alpine AS builder
WORKDIR /app

# Install dependencies using package-lock.json for consistent builds
COPY package*.json ./
RUN npm ci

# Copy source code and build the production bundle
COPY . .

# Set building environment variables. 
# NEXT_PUBLIC_API_URL is baked in during the build stage for client-side queries.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN npm run build

# Stage 2: Production runtime server environment
FROM node:18-alpine AS runner
WORKDIR /app

# Set production environment variable
ENV NODE_ENV=production

# Copy only the compiled outputs and required production dependencies
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public

# Expose Next.js port
EXPOSE 3000

# Start the Next.js production server
CMD ["npm", "start"]
