import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "InfraBench — GPU Infrastructure Planner",
  description:
    "Find the optimal GPU architecture for LLM inference and fine-tuning",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased min-h-screen">{children}</body>
    </html>
  );
}
