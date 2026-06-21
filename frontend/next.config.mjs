/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'https://vivo777-astram-api.hf.space/:path*',
      },
    ];
  },
};

export default nextConfig;

