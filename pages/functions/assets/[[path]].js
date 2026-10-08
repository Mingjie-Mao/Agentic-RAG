import { proxy } from "../../proxy.js";

export const onRequest = (context) => proxy(context);
