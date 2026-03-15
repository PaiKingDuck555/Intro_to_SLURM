/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverComponentsExternalPackages: ["ssh2"],
  },
};

module.exports = nextConfig;
