/** @type {import('next').NextConfig} */
const repo = "sopheakvoatei-seksopheakvoatei-sek.github.io";

const nextConfig = {
  output: "export",
  basePath: `/${repo}`,
  assetPrefix: `/${repo}/`,
  trailingSlash: true,

  images: {
    unoptimized: true,
  },
};

export default nextConfig;