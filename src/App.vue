<script setup lang="ts"></script>

<template>
  <Suspense>
    <router-view />
    <template #fallback> Loading... </template>
  </Suspense>
</template>

<style scoped lang="scss"></style>
