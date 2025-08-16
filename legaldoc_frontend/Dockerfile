# --------------------------
# Base image
# --------------------------
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package.json and lock files first
COPY package*.json ./

# Install dependencies
RUN npm install --production=false

# Copy the rest of the code
COPY . .

# Build the frontend (creates /app/dist)
RUN npm run build

# Expose port (Vite preview runs on 3000 by default)
EXPOSE 3000

# Run the built app with vite preview
CMD ["npm", "run", "preview", "--", "--host", "0.0.0.0"]
